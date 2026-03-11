#!/usr/bin/env ruby
# frozen_string_literal: true

# sync_docs_discourse.rb
#
# Syncs sunnypilot documentation from docs_sp/ to a Discourse forum.
# Reads raw .md files, converts MkDocs Material syntax to Discourse-compatible
# markdown (Obsidian-style callouts), resolves internal links to Discourse URLs,
# and creates or updates topics via the Discourse API.
#
# Usage:
#   ruby sync_docs_discourse.rb [--dry-run] [--verbose]
#
# Environment variables (required unless --dry-run):
#   DISCOURSE_URL      - Base URL of the Discourse instance (e.g. https://forum.sunnypilot.ai)
#   DISCOURSE_API_KEY  - API key with topic create/update permissions
#   DISCOURSE_API_USER - Username for API requests (e.g. system or a bot account)
#   DISCOURSE_CATEGORY - Category slug or ID for documentation topics (default: "documentation")
#
# Optional:
#   DOCS_BASE_URL      - Base URL for the static docs site (default: https://docs.sunnypilot.ai)

require "net/http"
require "uri"
require "json"
require "yaml"
require "digest"
require "fileutils"
require "optparse"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DOCS_DIR       = File.expand_path("../../", __FILE__)  # docs_sp/
MKDOCS_YML     = File.expand_path("../../../mkdocs-sp.yml", __FILE__)
CACHE_DIR      = File.expand_path("../../../.discourse_sync_cache", __FILE__)
DOCS_BASE_URL  = ENV.fetch("DOCS_BASE_URL", "https://docs.sunnypilot.ai")

DISCOURSE_URL      = ENV["DISCOURSE_URL"]
DISCOURSE_API_KEY  = ENV["DISCOURSE_API_KEY"]
DISCOURSE_API_USER = ENV.fetch("DISCOURSE_API_USER", "system")
DISCOURSE_CATEGORY = ENV.fetch("DISCOURSE_CATEGORY", "documentation")

# MkDocs admonition type → Obsidian/Discourse callout type
ADMONITION_MAP = {
  "note"      => "NOTE",
  "abstract"  => "ABSTRACT",
  "info"      => "INFO",
  "tip"       => "TIP",
  "success"   => "SUCCESS",
  "question"  => "QUESTION",
  "warning"   => "WARNING",
  "failure"   => "FAILURE",
  "danger"    => "DANGER",
  "bug"       => "BUG",
  "example"   => "EXAMPLE",
  "quote"     => "QUOTE",
}.freeze

# Pages to skip (not meaningful as standalone Discourse topics)
SKIP_FILES = %w[
  index.md
  README.md
].freeze

# ---------------------------------------------------------------------------
# CLI Options
# ---------------------------------------------------------------------------

options = { dry_run: false, verbose: false }
OptionParser.new do |opts|
  opts.banner = "Usage: #{$PROGRAM_NAME} [options]"
  opts.on("--dry-run", "Show what would be synced without making API calls") { options[:dry_run] = true }
  opts.on("--verbose", "Print detailed conversion output") { options[:verbose] = true }
  opts.on("-h", "--help", "Show this help") { puts opts; exit }
end.parse!

DRY_RUN = options[:dry_run]
VERBOSE = options[:verbose]

unless DRY_RUN
  %w[DISCOURSE_URL DISCOURSE_API_KEY].each do |var|
    abort "Error: #{var} environment variable is required" unless ENV[var]
  end
end

# ---------------------------------------------------------------------------
# MkDocs → Discourse Markdown Converter
# ---------------------------------------------------------------------------

module MkDocsConverter
  module_function

  # Main entry point: convert a full MkDocs Material markdown string
  # to Discourse-compatible markdown.
  def convert(content, file_path:, nav_slug_map: {})
    result = content.dup

    # 1. Strip YAML front matter
    result = strip_front_matter(result)

    # 2. Convert admonitions (!!! type "title" / ??? type "title")
    result = convert_admonitions(result)

    # 3. Convert Material tabs (=== "Tab Name")
    result = convert_tabs(result)

    # 4. Convert grid cards
    result = convert_grid_cards(result)

    # 5. Convert Material emoji shortcodes to Unicode or strip
    result = convert_emoji_shortcodes(result)

    # 6. Resolve internal .md links to docs site URLs
    result = resolve_internal_links(result, file_path: file_path)

    # 7. Clean up excessive blank lines
    result = result.gsub(/\n{4,}/, "\n\n\n")

    result.strip + "\n"
  end

  # Remove YAML front matter (--- ... ---)
  def strip_front_matter(content)
    content.sub(/\A---\n.*?\n---\n*/m, "")
  end

  # Convert MkDocs admonitions to Obsidian/Discourse callouts.
  #
  # Input:
  #   !!! warning "Title"
  #       Content line 1
  #       Content line 2
  #
  # Output:
  #   > [!WARNING] Title
  #   > Content line 1
  #   > Content line 2
  #
  # Also handles collapsible (??? / ???+) variants.
  def convert_admonitions(content)
    lines = content.lines
    result = []
    i = 0

    while i < lines.length
      line = lines[i]

      # Match admonition opener: !!! type "title" or ??? type "title" or ???+ type "title"
      if line =~ /^(\s*)(\!{3}|\?{3}\+?) (\w+)(?: "([^"]*)")?/
        indent = $1
        marker = $2
        ad_type = $3.downcase
        title = $4

        callout_type = ADMONITION_MAP[ad_type] || ad_type.upcase

        # Build the callout header
        header = "#{indent}> [!#{callout_type}]"
        header += " #{title}" if title && !title.empty?

        # For collapsible (???), add a note
        if marker.start_with?("???")
          collapsed = !marker.include?("+")
          header += " *(click to #{collapsed ? 'expand' : 'collapse'})*" if title.nil? || title.empty?
        end

        result << header + "\n"
        i += 1

        # Collect indented content lines (4 spaces deeper than the opener)
        content_indent = indent + "    "
        while i < lines.length
          content_line = lines[i]

          if content_line =~ /^#{Regexp.escape(content_indent)}/
            # Indented content line — part of the admonition
            stripped = content_line.sub(/^#{Regexp.escape(content_indent)}/, "")
            result << "#{indent}> #{stripped}"
            i += 1
          elsif content_line.strip.empty?
            # Blank line: only part of admonition if the next non-blank
            # line is still indented at content level
            j = i + 1
            j += 1 while j < lines.length && lines[j].strip.empty?
            if j < lines.length && lines[j] =~ /^#{Regexp.escape(content_indent)}/
              result << "#{indent}>\n"
              i += 1
            else
              # Blank line ends the admonition
              break
            end
          else
            break
          end
        end
      else
        result << line
        i += 1
      end
    end

    result.join
  end

  # Convert Material tabs to Discourse-friendly headings with horizontal rules.
  #
  # Input:
  #   === "Tab Name"
  #       Content
  #
  # Output:
  #   **Tab Name**
  #
  #   Content
  #
  #   ---
  def convert_tabs(content)
    lines = content.lines
    result = []
    i = 0

    while i < lines.length
      line = lines[i]

      if line =~ /^(\s*)=== "([^"]+)"/
        indent = $1
        tab_name = $2

        result << "#{indent}**#{tab_name}**\n"
        result << "\n"
        i += 1

        # Collect indented content
        content_indent = indent + "    "
        while i < lines.length
          content_line = lines[i]
          if content_line =~ /^#{Regexp.escape(content_indent)}/ || content_line.strip.empty?
            if content_line.strip.empty?
              result << "\n"
            else
              stripped = content_line.sub(/^#{Regexp.escape(content_indent)}/, "")
              result << "#{indent}#{stripped}"
            end
            i += 1
          else
            break
          end
        end

        result << "#{indent}---\n"
        result << "\n"
      else
        result << line
        i += 1
      end
    end

    result.join
  end

  # Convert grid cards to simple lists (Discourse doesn't support grid cards)
  def convert_grid_cards(content)
    content
      .gsub(/<div class="grid cards" markdown>/, "")
      .gsub(/<\/div>/, "")
  end

  # Convert Material emoji shortcodes to plain text or remove
  # e.g., :material-rocket-launch: → 🚀 (or just strip)
  EMOJI_MAP = {
    ":material-rocket-launch:" => "🚀",
    ":material-cog:" => "⚙️",
    ":material-car:" => "🚗",
    ":material-shield:" => "🛡️",
  }.freeze

  def convert_emoji_shortcodes(content)
    result = content.dup
    EMOJI_MAP.each do |shortcode, emoji|
      result.gsub!(shortcode, emoji)
    end
    # Strip any remaining :material-*: shortcodes
    result.gsub(/:material-[\w-]+:/, "")
  end

  # Resolve internal links: (../features/icbm.md) → (https://docs.sunnypilot.ai/features/icbm/)
  def resolve_internal_links(content, file_path:)
    content.gsub(/\]\(([^)]+\.md)\)/) do |match|
      relative_path = $1
      # Skip external URLs
      next match if relative_path.start_with?("http")

      # Resolve the relative path from the current file's directory
      current_dir = File.dirname(file_path)
      resolved = File.expand_path(relative_path, current_dir)

      # Make it relative to docs_sp/
      docs_relative = resolved.sub(%r{^.*/docs_sp/}, "")

      # Convert to URL path: remove .md, add trailing slash
      url_path = docs_relative.sub(/\.md$/, "/")
      "](#{DOCS_BASE_URL}/#{url_path})"
    end
  end
end

# ---------------------------------------------------------------------------
# Discourse API Client
# ---------------------------------------------------------------------------

module DiscourseAPI
  module_function

  def base_uri
    URI.parse(DISCOURSE_URL)
  end

  def headers
    {
      "Content-Type" => "application/json",
      "Api-Key" => DISCOURSE_API_KEY,
      "Api-Username" => DISCOURSE_API_USER,
    }
  end

  # Look up category ID by slug
  def category_id(slug)
    uri = URI.join(DISCOURSE_URL, "/c/#{slug}/show.json")
    response = http_get(uri)
    return nil unless response.is_a?(Net::HTTPSuccess)

    data = JSON.parse(response.body)
    data.dig("category", "id")
  end

  # Search for an existing topic by external_id (stored in the topic's first post)
  # We use a convention: embed an HTML comment <!-- docs-sync-id: path/to/file.md -->
  def find_topic_by_sync_id(sync_id)
    search_query = "<!-- docs-sync-id: #{sync_id} -->"
    uri = URI.join(DISCOURSE_URL, "/search.json?q=#{URI.encode_www_form_component(search_query)}")
    response = http_get(uri)
    return nil unless response.is_a?(Net::HTTPSuccess)

    data = JSON.parse(response.body)
    topics = data.dig("topics") || []
    topics.first
  end

  # Create a new topic
  def create_topic(title:, raw:, category_id:, tags: [])
    uri = URI.join(DISCOURSE_URL, "/posts.json")
    payload = {
      title: title,
      raw: raw,
      category: category_id,
      tags: tags,
    }
    http_post(uri, payload)
  end

  # Update an existing topic's first post
  def update_post(post_id:, raw:, edit_reason: "Documentation sync")
    uri = URI.join(DISCOURSE_URL, "/posts/#{post_id}.json")
    payload = {
      post: {
        raw: raw,
        edit_reason: edit_reason,
      },
    }
    http_put(uri, payload)
  end

  # Get a topic's first post ID
  def first_post_id(topic_id)
    uri = URI.join(DISCOURSE_URL, "/t/#{topic_id}.json")
    response = http_get(uri)
    return nil unless response.is_a?(Net::HTTPSuccess)

    data = JSON.parse(response.body)
    data.dig("post_stream", "posts", 0, "id")
  end

  # --- HTTP helpers ---

  def http_get(uri)
    http = Net::HTTP.new(uri.host, uri.port)
    http.use_ssl = uri.scheme == "https"
    request = Net::HTTP::Get.new(uri, headers)
    http.request(request)
  end

  def http_post(uri, payload)
    http = Net::HTTP.new(uri.host, uri.port)
    http.use_ssl = uri.scheme == "https"
    request = Net::HTTP::Post.new(uri, headers)
    request.body = payload.to_json
    http.request(request)
  end

  def http_put(uri, payload)
    http = Net::HTTP.new(uri.host, uri.port)
    http.use_ssl = uri.scheme == "https"
    request = Net::HTTP::Put.new(uri, headers)
    request.body = payload.to_json
    http.request(request)
  end
end

# ---------------------------------------------------------------------------
# Nav Parser — extract title + path from mkdocs-sp.yml nav
# ---------------------------------------------------------------------------

module NavParser
  module_function

  # Parse the mkdocs nav structure into a flat list of { title:, path: }
  def parse(mkdocs_yml_path)
    config = YAML.safe_load(File.read(mkdocs_yml_path))
    nav = config["nav"] || []
    flatten_nav(nav)
  end

  def flatten_nav(items, prefix_parts = [])
    result = []
    items.each do |item|
      case item
      when Hash
        item.each do |key, value|
          case value
          when String
            # Skip external links
            next if value.start_with?("http")
            result << { title: key, path: value, breadcrumb: prefix_parts + [key] }
          when Array
            result.concat(flatten_nav(value, prefix_parts + [key]))
          end
        end
      when String
        # Bare path without title (unlikely in our nav)
        result << { title: File.basename(item, ".md").tr("-", " ").capitalize, path: item }
      end
    end
    result
  end
end

# ---------------------------------------------------------------------------
# Content Cache — skip unchanged files
# ---------------------------------------------------------------------------

module ContentCache
  module_function

  def cache_path(file_path)
    slug = file_path.gsub("/", "_").gsub(".md", "")
    File.join(CACHE_DIR, "#{slug}.sha256")
  end

  def changed?(file_path, content_hash)
    cached = cache_path(file_path)
    return true unless File.exist?(cached)
    File.read(cached).strip != content_hash
  end

  def save(file_path, content_hash)
    FileUtils.mkdir_p(CACHE_DIR)
    File.write(cache_path(file_path), content_hash)
  end
end

# ---------------------------------------------------------------------------
# Main Sync Logic
# ---------------------------------------------------------------------------

def sync_doc(entry, category_id)
  file_path = File.join(DOCS_DIR, entry[:path])
  unless File.exist?(file_path)
    puts "  ⚠ File not found: #{file_path}"
    return :skipped
  end

  raw_content = File.read(file_path, encoding: "utf-8")
  content_hash = Digest::SHA256.hexdigest(raw_content)

  # Skip unchanged files
  unless ContentCache.changed?(entry[:path], content_hash)
    puts "  ✓ Unchanged: #{entry[:path]}" if VERBOSE
    return :unchanged
  end

  # Convert MkDocs → Discourse markdown
  converted = MkDocsConverter.convert(raw_content, file_path: file_path)

  # Prepend breadcrumb navigation
  if entry[:breadcrumb] && entry[:breadcrumb].length > 1
    breadcrumb = entry[:breadcrumb][0..-2].join(" › ")
    converted = "*#{breadcrumb}*\n\n#{converted}"
  end

  # Append sync ID comment and docs site link
  docs_url = "#{DOCS_BASE_URL}/#{entry[:path].sub(/\.md$/, '/')}"
  converted += "\n\n---\n"
  converted += "*This documentation is automatically synced from the [sunnypilot docs site](#{docs_url}).*\n"
  converted += "<!-- docs-sync-id: #{entry[:path]} -->\n"

  title = entry[:title]

  if DRY_RUN
    puts "  → Would sync: \"#{title}\" (#{entry[:path]})"
    if VERBOSE
      puts "    --- Converted content (first 500 chars) ---"
      puts "    #{converted[0..500].gsub("\n", "\n    ")}"
      puts "    ---"
    end
    ContentCache.save(entry[:path], content_hash) # Cache even in dry-run for testing
    return :would_sync
  end

  # Check if topic already exists
  existing = DiscourseAPI.find_topic_by_sync_id(entry[:path])

  if existing
    # Update existing topic
    post_id = DiscourseAPI.first_post_id(existing["id"])
    if post_id
      response = DiscourseAPI.update_post(post_id: post_id, raw: converted)
      if response.is_a?(Net::HTTPSuccess)
        puts "  ✓ Updated: \"#{title}\" (topic ##{existing['id']})"
        ContentCache.save(entry[:path], content_hash)
        return :updated
      else
        puts "  ✗ Failed to update: #{response.code} #{response.body[0..200]}"
        return :error
      end
    else
      puts "  ✗ Could not find first post for topic ##{existing['id']}"
      return :error
    end
  else
    # Create new topic
    response = DiscourseAPI.create_topic(
      title: "#{title} — sunnypilot Docs",
      raw: converted,
      category_id: category_id,
      tags: ["docs", "auto-sync"]
    )
    if response.is_a?(Net::HTTPSuccess) || response.is_a?(Net::HTTPCreated)
      data = JSON.parse(response.body)
      puts "  ✓ Created: \"#{title}\" (topic ##{data['topic_id']})"
      ContentCache.save(entry[:path], content_hash)
      return :created
    else
      puts "  ✗ Failed to create: #{response.code} #{response.body[0..200]}"
      return :error
    end
  end
end

# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main
  puts "sunnypilot Documentation → Discourse Sync"
  puts "=" * 50
  puts "Mode: #{DRY_RUN ? 'DRY RUN' : 'LIVE'}"
  puts "Docs dir: #{DOCS_DIR}"
  puts "Discourse: #{DISCOURSE_URL || '(dry-run, no URL)'}"
  puts

  # Parse nav entries
  nav_entries = NavParser.parse(MKDOCS_YML)
  puts "Found #{nav_entries.length} nav entries"

  # Filter out skipped files
  nav_entries.reject! { |e| SKIP_FILES.include?(File.basename(e[:path])) }
  puts "After filtering: #{nav_entries.length} pages to sync"
  puts

  # Resolve category ID
  category_id = nil
  unless DRY_RUN
    category_id = DiscourseAPI.category_id(DISCOURSE_CATEGORY)
    abort "Error: Could not find category '#{DISCOURSE_CATEGORY}'" unless category_id
    puts "Category: #{DISCOURSE_CATEGORY} (ID: #{category_id})"
    puts
  end

  # Sync each page
  stats = { created: 0, updated: 0, unchanged: 0, skipped: 0, error: 0, would_sync: 0 }
  nav_entries.each_with_index do |entry, idx|
    # Rate limiting: 1 request per second for live mode
    sleep(1) if !DRY_RUN && idx > 0

    result = sync_doc(entry, category_id)
    stats[result] = (stats[result] || 0) + 1
  end

  # Summary
  puts
  puts "=" * 50
  puts "Sync complete!"
  stats.each do |status, count|
    next if count == 0
    puts "  #{status}: #{count}"
  end
end

main
